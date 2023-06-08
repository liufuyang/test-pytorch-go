package main

import (
	"fmt"
	"log"

	"github.com/sugarme/gotch/ts"
)

func main() {

	// Load the Python saved module.
	model, err := ts.ModuleLoad("../python/model_scripted.pt")
	if err != nil {
		log.Fatal(err)
	}

	inputTensor := ts.TensorFrom([]float32{4.0})
	forward, err := model.Forward(inputTensor)
	if err != nil {
		fmt.Println(err.Error())
		return
	}
	fmt.Printf("predict: %8.3f\n", forward.MustFloat64Value([]int64{0}))
	fmt.Printf("%i", forward)

}
